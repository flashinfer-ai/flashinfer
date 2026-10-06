"""Per-peer RoCE route planning: switched fabrics keep index pairing, switchless rings pair by subnet."""

import ipaddress

import pytest

from b12x.comm.roce._routes import Endpoint, plan_routes


def ep(name, iface=None):
    """Endpoint named ``name`` with an optional IPv4 interface such as ``10.0.0.1/30``."""
    return Endpoint(name, None if iface is None else ipaddress.IPv4Interface(iface))


# Three DGX Sparks cabled port 0 -> next node's port 1 (no switch). Every cable shows up as two
# PCIe functions, each with its own /30, so each peer is reachable over exactly two local devices.
RING = [
    [ep("rocep1s0f0", "10.10.0.1/30"), ep("rocep1s0f1", "10.10.4.2/30"),
     ep("roceP2p1s0f0", "10.10.1.1/30"), ep("roceP2p1s0f1", "10.10.5.2/30")],
    [ep("rocep1s0f0", "10.10.2.1/30"), ep("rocep1s0f1", "10.10.0.2/30"),
     ep("roceP2p1s0f0", "10.10.3.1/30"), ep("roceP2p1s0f1", "10.10.1.2/30")],
    [ep("rocep1s0f0", "10.10.4.1/30"), ep("rocep1s0f1", "10.10.2.2/30"),
     ep("roceP2p1s0f0", "10.10.5.1/30"), ep("roceP2p1s0f1", "10.10.3.2/30")],
]


def _link(endpoints, rank, peer, local, remote):
    """True when the two chosen devices share an IPv4 network."""
    a = endpoints[rank][local].ipv4
    b = endpoints[peer][remote].ipv4
    return a.network == b.network


def test_ring_routes_every_rail_over_a_real_link():
    """Every rail to every peer uses two devices on the same /30, and a peer's rails use distinct devices."""
    rails = 2
    for rank in range(3):
        routes = plan_routes(RING, rank, rails)
        assert routes[rank] == []
        for peer in range(3):
            if peer == rank:
                continue
            assert len(routes[peer]) == rails
            assert len({l for l, _ in routes[peer]}) == rails
            for local, remote in routes[peer]:
                assert _link(RING, rank, peer, local, remote)


def test_ring_routes_are_symmetric_so_both_ends_use_the_same_link_per_rail():
    """Rank a's route to b is rank b's route to a with the ends swapped, rail for rail."""
    plans = [plan_routes(RING, rank, 2) for rank in range(3)]
    for a in range(3):
        for b in range(3):
            if a != b:
                assert plans[a][b] == [(r, l) for l, r in plans[b][a]]


def test_ring_rank0_uses_port0_functions_to_the_next_node_and_port1_functions_to_the_previous():
    """On the ring, port 0's two functions reach the next node and port 1's reach the previous one."""
    routes = plan_routes(RING, 0, 2)
    names = RING[0]
    assert {names[l].name for l, _ in routes[1]} == {"rocep1s0f0", "roceP2p1s0f0"}
    assert {names[l].name for l, _ in routes[2]} == {"rocep1s0f1", "roceP2p1s0f1"}


def test_switched_fabric_keeps_index_pairing():
    """Two subnets shared by every rank: rail h uses device h at both ends."""
    switched = [
        [ep("rocep1s0f0", "192.168.42.10/24"), ep("roceP2p1s0f0", "192.168.43.10/24")],
        [ep("rocep1s0f0", "192.168.42.11/24"), ep("roceP2p1s0f0", "192.168.43.11/24")],
        [ep("rocep1s0f0", "192.168.42.12/24"), ep("roceP2p1s0f0", "192.168.43.12/24")],
    ]
    for rank in range(3):
        routes = plan_routes(switched, rank, 2)
        for peer in range(3):
            if peer != rank:
                assert routes[peer] == [(0, 0), (1, 1)]


def test_one_flat_subnet_with_extra_devices_keeps_index_pairing_on_the_first_rails():
    """Four devices on one subnet: the rails use devices 0 and 1 at both ends."""
    flat = [[ep(f"d{h}", f"10.0.0.{10 * r + h}/24") for h in range(4)] for r in range(2)]
    assert plan_routes(flat, 0, 2)[1] == [(0, 0), (1, 1)]


def test_unknown_addresses_fall_back_to_index_pairing():
    """Devices without IPv4 GIDs are paired in the order given."""
    unknown = [[ep("mlx5_0"), ep("mlx5_1")], [ep("mlx5_0"), ep("mlx5_1")]]
    assert plan_routes(unknown, 1, 2)[0] == [(0, 0), (1, 1)]


def test_two_node_direct_cable_pairs_the_matching_functions():
    """A two-node direct cable pairs each function with the one on its subnet, whatever the device order."""
    direct = [
        [ep("rocep1s0f0", "10.20.0.1/30"), ep("roceP2p1s0f0", "10.20.1.1/30")],
        [ep("roceP2p1s0f0", "10.20.1.2/30"), ep("rocep1s0f0", "10.20.0.2/30")],
    ]
    assert plan_routes(direct, 0, 2)[1] == [(0, 1), (1, 0)]
    assert plan_routes(direct, 1, 2)[0] == [(1, 0), (0, 1)]


def test_single_rail_uses_one_link_per_peer():
    """With one rail, each peer gets one route over a real link."""
    routes = plan_routes(RING, 1, 1)
    for peer in (0, 2):
        (local, remote), = routes[peer]
        assert _link(RING, 1, peer, local, remote)


def test_peer_with_too_few_links_is_a_clear_error():
    """A peer reachable over fewer links than rails fails with the ranks and counts named."""
    broken = [list(r) for r in RING]
    # Rank 2 loses roceP2p1s0f1 (10.10.3.2), one of its two links to rank 1.
    broken[2] = [e for e in RING[2] if e.name != "roceP2p1s0f1"]
    with pytest.raises(RuntimeError, match="rank 1: 1 of 2 .* rank 2"):
        plan_routes(broken, 1, 2)


def test_fallback_keeps_an_unverifiable_same_index_pair_after_verified_links():
    """Subnet fallback still trusts a same-index pair it cannot check, but only after verified links."""
    mixed = [
        [ep("ib0"), ep("roce1", "10.0.0.1/30"), ep("roce2", "10.0.9.1/30")],
        [ep("ib0"), ep("roce1", "10.0.5.2/30"), ep("roce2", "10.0.0.2/30")],
    ]
    # Device 1 pairs with device 1 on different subnets, so index pairing is invalid; the only
    # verified link is rank 0 device 1 to rank 1 device 2, and device 0 on both ends has no IPv4 GID.
    assert plan_routes(mixed, 0, 2)[1] == [(1, 2), (0, 0)]
    assert plan_routes(mixed, 1, 2)[0] == [(2, 1), (0, 0)]


def test_fallback_prefers_verified_links_over_unverifiable_pairs():
    """With enough verified links, an unverifiable same-index pair is not used."""
    ring = [list(r) for r in RING]
    ring[0] = [ep("mlx5_0")] + ring[0]
    ring[1] = [ep("mlx5_0")] + ring[1]
    ring[2] = [ep("mlx5_0")] + ring[2]
    for rank in range(3):
        for peer, links in enumerate(plan_routes(ring, rank, 2)):
            assert (0, 0) not in links


def test_fallback_finds_a_complete_matching_a_greedy_pick_would_miss():
    """A verified link that blocks the only unverifiable pair is skipped for one that leaves room for it."""
    local = [ep("roce0", "10.0.0.1/24"), ep("ib1")]
    remote = [ep("roce0", "10.0.9.2/24"), ep("roce1", "10.0.0.2/24"), ep("roce2", "10.0.0.3/24")]
    # Taking (0, 1) first would leave (1, 1) unusable; (0, 2) plus the trusted (1, 1) supplies both rails.
    assert plan_routes([local, remote], 0, 2)[1] == [(0, 2), (1, 1)]
    assert plan_routes([local, remote], 1, 2)[0] == [(2, 0), (1, 1)]


def test_fallback_tie_break_is_symmetric_on_a_shared_subnet():
    """Two equally verified matchings on one flat subnet: both ends still choose the same links."""
    a = [ep("d0", "10.1.0.1/24"), ep("d1", "10.1.0.2/24"), ep("x", "10.9.0.1/24")]
    b = [ep("d0", "10.1.0.5/24"), ep("d1", "10.1.0.6/24"), ep("y", "10.8.0.1/24")]
    # Index pairing fails on device 2, so both ranks fall back to subnet matching.
    forward = plan_routes([a, b], 0, 2)[1]
    backward = plan_routes([a, b], 1, 2)[0]
    assert backward == [(r, l) for l, r in forward]


def test_rails_out_of_range_rejected():
    """Rail counts outside 1..MAX_RAILS are rejected."""
    with pytest.raises(ValueError):
        plan_routes(RING, 0, 0)
    with pytest.raises(ValueError):
        plan_routes(RING, 0, 3)


@pytest.mark.parametrize("gid,netmask,expected", [
    ("0000:0000:0000:0000:0000:ffff:0a0a:0001", "255.255.255.252", "10.10.0.1/30"),
    ("0000:0000:0000:0000:0000:ffff:0a0a:0001", None, None),
    ("fe80:0000:0000:0000:0000:0000:0000:0001", "255.255.255.252", None),
    ("invalid", "255.255.255.252", None),
])
def test_local_endpoints_uses_gid_netdev_prefix(tmp_path, monkeypatch, gid, netmask, expected):
    from b12x.comm.roce import _routes

    port = tmp_path / "rdma0" / "ports" / "1"
    (port / "gids").mkdir(parents=True)
    (port / "gid_attrs" / "ndevs").mkdir(parents=True)
    (port / "gids" / "3").write_text(gid)
    (port / "gid_attrs" / "ndevs" / "3").write_text("eth7")
    seen = []

    def mask(name):
        seen.append(name)
        return netmask

    monkeypatch.setattr(_routes, "_netmask", mask)
    actual = _routes.local_endpoints(("rdma0", "missing"), 3, root=tmp_path)
    assert actual == (ep("rdma0", expected), ep("missing"))
    assert seen == (["eth7"] if "ffff" in gid else [])
