"""Choose which local and remote RDMA device carries each stripe ("rail") to each peer.

On a switched fabric every device reaches every peer, so rail ``h`` simply pairs local device ``h``
with the peer's device ``h``. A switchless topology breaks that: three DGX Sparks cabled in a ring
(port 0 to the next node's port 1) reach each neighbour through one port only, and each port is two
PCIe functions with their own point-to-point subnet. There, rail ``h`` to a peer must use a local
device and a remote device on the same subnet.

``plan_routes`` uses index pairing (rail ``h`` on device ``h`` at both ends) whenever every pair of
ranks shares a subnet on each such pair, which every switched fabric does, and otherwise pairs devices
per peer by subnet. A device without an IPv4 GID cannot be checked, so both modes trust the caller's
device order for it: index pairing skips the check, and per-peer pairing uses a same-index pair with
such a device only when no choice of verified same-subnet links fills every rail. Per-peer pairing
picks the device-disjoint set of links with the most verified ones. Every rank runs the planner over
the same exchanged endpoint list, and rails are ordered by link network address, so both ends of a
link always agree on which rail it carries.
"""

from __future__ import annotations

import fcntl
import ipaddress
import itertools
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

MAX_RAILS = 2
_SIOCGIFNETMASK = 0x891B


@dataclass(frozen=True)
class Endpoint:
    """One local RDMA device: its name and, when its GID is IPv4-mapped, the address with prefix."""

    name: str
    ipv4: Optional[ipaddress.IPv4Interface]


def _netmask(ifname: str) -> Optional[str]:
    """IPv4 netmask of ``ifname`` (SIOCGIFNETMASK), or None when unavailable."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            raw = fcntl.ioctl(
                sock.fileno(),
                _SIOCGIFNETMASK,
                struct.pack("256s", ifname.encode()[:15]),
            )
        except OSError:
            return None
    return socket.inet_ntoa(raw[20:24])


def local_endpoints(
    names: Sequence[str], gid_index: int, root: Path = Path("/sys/class/infiniband")
) -> tuple[Endpoint, ...]:
    """Describe ``names``: the IPv4 address of each device's GID at ``gid_index`` and its netdev's prefix.

    A device whose GID is not IPv4-mapped, or whose netdev or netmask cannot be read, gets
    ``ipv4=None`` and is treated as reachable by index pairing only.
    """
    out = []
    for name in names:
        port = root / name / "ports" / "1"
        ipv4 = None
        try:
            raw = bytes.fromhex(
                port.joinpath("gids", str(gid_index))
                .read_text()
                .strip()
                .replace(":", "")
            )
            ifname = (
                port.joinpath("gid_attrs", "ndevs", str(gid_index)).read_text().strip()
            )
        except (OSError, ValueError):
            raw, ifname = b"", ""
        if (
            len(raw) == 16
            and raw[:10] == bytes(10)
            and raw[10:12] == b"\xff\xff"
            and ifname
        ):
            mask = _netmask(ifname)
            if mask is not None:
                ipv4 = ipaddress.IPv4Interface(f"{socket.inet_ntoa(raw[12:])}/{mask}")
        out.append(Endpoint(name, ipv4))
    return tuple(out)


def _same_link(a: Endpoint, b: Endpoint) -> bool:
    """True when both endpoints have IPv4 addresses on the same network."""
    return (
        a.ipv4 is not None and b.ipv4 is not None and a.ipv4.network == b.ipv4.network
    )


def _index_pairing_valid(endpoints: Sequence[Sequence[Endpoint]], rails: int) -> bool:
    """True when rail ``h`` can pair device ``h`` with device ``h`` between every pair of ranks."""
    if any(len(devices) < rails for devices in endpoints):
        return False
    for a in range(len(endpoints)):
        for b in range(a + 1, len(endpoints)):
            for h in range(rails):
                x, y = endpoints[a][h], endpoints[b][h]
                if x.ipv4 is None or y.ipv4 is None:
                    continue  # no IPv4 GID to compare: trust the caller's device order
                if x.ipv4.network != y.ipv4.network:
                    return False
    return True


def _best_matching(candidates: list, rails: int) -> Optional[list[tuple[int, int]]]:
    """The ``rails`` candidates with distinct local and remote devices that use the most verified links,
    ties broken by the smallest sorted keys; ``None`` when no such set exists.

    ``candidates`` is sorted by key, so each combination, and therefore the rail order, is ordered by key.
    A rank has at most four devices, so the search is at most C(16, 2) combinations.
    """
    best = None
    for combo in itertools.combinations(candidates, rails):
        if (
            len({l for _, l, _ in combo}) < rails
            or len({r for _, _, r in combo}) < rails
        ):
            continue
        score = (sum(key[0] for key, _, _ in combo), [key for key, _, _ in combo])
        if best is None or score < best[0]:
            best = (score, combo)
    return None if best is None else [(l, r) for _, l, r in best[1]]


def plan_routes(
    endpoints: Sequence[Sequence[Endpoint]], rank: int, rails: int
) -> list[list[tuple[int, int]]]:
    """Per peer, the ``(local_device, remote_device)`` index pair carrying each rail; ``[]`` for ``rank``."""
    if not 1 <= rails <= MAX_RAILS:
        raise ValueError(f"rails must be 1..{MAX_RAILS}, got {rails}")
    world = len(endpoints)
    if _index_pairing_valid(endpoints, rails):
        return [
            [] if p == rank else [(h, h) for h in range(rails)] for p in range(world)
        ]
    local = endpoints[rank]
    routes: list[list[tuple[int, int]]] = []
    for peer in range(world):
        if peer == rank:
            routes.append([])
            continue
        remote = endpoints[peer]
        # Candidates: verified same-subnet links, keyed by network and the link's two addresses; and
        # same-index pairs with no IPv4 GID on either end, which cannot be checked and are trusted in
        # caller order, as index pairing would. Every key is symmetric between the two ranks.
        verified = [
            (
                (
                    0,
                    int(local[l].ipv4.network.network_address),
                    min(int(local[l].ipv4.ip), int(remote[r].ipv4.ip)),
                    max(int(local[l].ipv4.ip), int(remote[r].ipv4.ip)),
                ),
                l,
                r,
            )
            for l in range(len(local))
            for r in range(len(remote))
            if _same_link(local[l], remote[r])
        ]
        unverifiable = [
            ((1, h, 0, 0), h, h)
            for h in range(min(len(local), len(remote)))
            if local[h].ipv4 is None or remote[h].ipv4 is None
        ]
        candidates = sorted(verified + unverifiable)
        chosen = _best_matching(candidates, rails)
        if chosen is None:
            reach = next(
                (k for k in range(rails - 1, 0, -1) if _best_matching(candidates, k)), 0
            )
            raise RuntimeError(
                f"rank {rank}: {reach} of {rails} RoCE rails have a link to rank {peer}; "
                f"local {[(e.name, str(e.ipv4)) for e in local]}, "
                f"remote {[(e.name, str(e.ipv4)) for e in remote]}"
            )
        routes.append(chosen)
    return routes


__all__ = ["Endpoint", "MAX_RAILS", "local_endpoints", "plan_routes"]
