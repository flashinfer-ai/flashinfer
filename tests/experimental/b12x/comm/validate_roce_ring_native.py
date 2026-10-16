"""Exercise the real RDMA ring proxy without a CUDA context.

Run one process per host with --rank R and --world-size N and --rendezvous HOST:PORT. The
CPU drives the same pinned-region doorbells and checks every received byte.
This checks transport progress and staging reuse, independently of GPU kernels.
"""

import argparse
import ctypes
import importlib.util
import json
import mmap
from pathlib import Path
import socket
import struct
import sys
import time


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def send(sock, data):
    sock.sendall(struct.pack("!I", len(data)) + data)


def receive(sock):
    def exact(n):
        data = bytearray()
        while len(data) < n:
            part = sock.recv(n - len(data))
            if not part:
                raise RuntimeError("peer disconnected")
            data += part
        return bytes(data)

    return exact(struct.unpack("!I", exact(4))[0])


class Group:
    def __init__(self, rank, address, world):
        self.rank, self.world = rank, world
        host, port = address.rsplit(":", 1)
        if rank == 0:
            with socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("0.0.0.0", int(port)))
                listener.listen(world - 1)
                listener.settimeout(30)
                self.peers = {}
                for _ in range(world - 1):
                    peer, _ = listener.accept()
                    peer.settimeout(30)
                    self.peers[int(receive(peer))] = peer
        else:
            until = time.monotonic() + 30
            while True:
                try:
                    self.peer = socket.create_connection((host, int(port)), 5)
                    break
                except OSError:
                    if time.monotonic() > until:
                        raise
                    time.sleep(0.1)
            self.peer.settimeout(30)
            send(self.peer, str(rank).encode())

    def exchange(self, value):
        if self.rank:
            send(self.peer, json.dumps(value).encode())
            return json.loads(receive(self.peer))
        values = [value] + [
            json.loads(receive(self.peers[r])) for r in range(1, self.world)
        ]
        for peer in self.peers.values():
            send(peer, json.dumps(values).encode())
        return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4, choices=range(2, 17))
    parser.add_argument("--rendezvous", required=True)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="check the ordinary direct transport instead",
    )
    parser.add_argument("--skew-rank", type=int)
    parser.add_argument("--hca-names", default="roceP2p1s0f0,rocep1s0f0")
    parser.add_argument("--gid-index", type=int, default=3)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.source is None:
        args.source = (
            Path(__file__).resolve().parents[4]
            / "flashinfer/experimental/b12x/comm/roce"
        )
    world = args.world_size
    if not 0 <= args.rank < world:
        parser.error("--rank must be less than --world-size")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    names = tuple(args.hca_names.split(","))
    if not 2 <= len(names) <= 4 or not all(names):
        parser.error("this validator requires 2..4 --hca-names for two rails")
    proxy_api = module(args.source / "_proxy.py", "ring_proxy_check")
    routes_api = module(args.source / "_routes.py", "ring_routes_check")
    group = Group(args.rank, args.rendezvous, world)
    layout = proxy_api.Layout(world, 1 << 20, 4 << 20, 1 << 20)
    memory = mmap.mmap(-1, layout.total_bytes)
    buffer = (ctypes.c_char * layout.total_bytes).from_buffer(memory)
    address = ctypes.addressof(buffer)
    proxy = proxy_api.Proxy(
        world_size=world,
        rank=args.rank,
        hca_names=names,
        gid_index=args.gid_index,
        region_ptr=address,
        region_bytes=layout.total_bytes,
        slot_bytes=1 << 20,
        pipe_bytes=4 << 20,
        varlen_bytes=1 << 20,
        ring=not args.mesh,
    )
    endpoints = routes_api.local_endpoints(names, args.gid_index)
    local = {
        "blob": proxy.local_blob().hex(),
        "endpoints": [(e.name, str(e.ipv4)) for e in endpoints],
    }
    records = group.exchange(local)
    import ipaddress

    all_endpoints = [
        [
            routes_api.Endpoint(n, ipaddress.IPv4Interface(ip))
            for n, ip in record["endpoints"]
        ]
        for record in records
    ]
    routes = routes_api.plan_routes(all_endpoints, args.rank, 2, ring=not args.mesh)
    if not args.mesh:
        assert all(
            not links
            for peer, links in enumerate(routes)
            if peer not in {args.rank, (args.rank + 1) % world, (args.rank - 1) % world}
        )
    proxy.connect([bytes.fromhex(r["blob"]) for r in records], routes)
    group.exchange("connected")
    proxy.start()
    ctrl = (ctypes.c_uint32 * 1024).from_address(address + layout.ctrl_off)
    # libatomic provides system acquire/release accesses on ARM as well as x86.
    atomics = ctypes.CDLL("libatomic.so.1")
    atomics.__atomic_load_4.argtypes = [ctypes.c_void_p, ctypes.c_int]
    atomics.__atomic_load_4.restype = ctypes.c_uint32
    atomics.__atomic_store_4.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]

    def put(offset, value):
        atomics.__atomic_store_4(address + offset, value, 3)

    def wait(offset, value):
        until = time.monotonic() + 20
        while atomics.__atomic_load_4(address + offset, 2) != value:
            if proxy.failed():
                raise RuntimeError(proxy.error())
            if time.monotonic() > until:
                raise TimeoutError(
                    f"rank {args.rank}: offset {offset} expected {value}, stats {proxy.stats()}"
                )

    try:
        for seq in range(1, args.iterations + 1):
            if args.rank == args.skew_rank:
                time.sleep(0.001 * (seq % 4))
            mode = (seq - 1) % 3
            slot = seq & 1
            pattern = (seq + 17 * args.rank) % 256
            if mode == 0:
                size = (32, 32768, 131072, 1 << 20)[((seq - 1) // 3) % 4]
                start = layout.send_off + slot * (1 << 20)
                memory[start : start + size] = bytes([pattern]) * size
                ctrl[4 + slot], ctrl[8 + slot] = size, 0
                put(layout.ctrl_off, seq)
                for src in range(world):
                    if src == args.rank:
                        continue
                    for rail in range(2):
                        wait(
                            layout.flag_off
                            + ((src * 2 + slot) * 2 + rail) * layout.flag_stride,
                            seq,
                        )
                    start = layout.recv_off + (src * 2 + slot) * (1 << 20)
                    assert (
                        memory[start : start + size]
                        == bytes([(seq + 17 * src) % 256]) * size
                    )
            elif mode == 1:
                size = (2 << 20, 4 << 20, (1 << 20) + 48)[((seq - 1) // 3) % 3]
                total_packs = size // 16
                chunk_bytes = 65536
                chunks = (
                    (total_packs + world - 1) // world * 16 + chunk_bytes - 1
                ) // chunk_bytes
                start = layout.pipe_send_off
                memory[start : start + size] = bytes([pattern]) * size
                ctrl[4 + slot], ctrl[8 + slot] = size, 1
                ctrl[10 + slot], ctrl[12 + slot] = chunk_bytes // 16, chunks
                for chunk in range(chunks):
                    put(layout.ctrl_off + 4 * (64 + chunk), seq)
                put(layout.ctrl_off, seq)
                first = args.rank * total_packs // world * 16
                end = (args.rank + 1) * total_packs // world * 16
                reduced = (world * seq + 17 * world * (world - 1) // 2) % 256
                for chunk in range(chunks):
                    length = min(chunk_bytes, max(0, end - first - chunk * chunk_bytes))
                    for src in range(world):
                        if src == args.rank:
                            continue
                        for rail in range(2):
                            wait(
                                layout.pipe_flag_off
                                + ((src * 64 + chunk) * 2 + rail) * layout.flag_stride,
                                seq,
                            )
                        start = (
                            layout.pipe_recv_off
                            + src * 2 * layout.shard_cap
                            + chunk * chunk_bytes
                        )
                        assert (
                            memory[start : start + length]
                            == bytes([(seq + 17 * src) % 256]) * length
                        )
                    start = layout.pipe_send_off + first + chunk * chunk_bytes
                    memory[start : start + length] = bytes([reduced]) * length
                    put(layout.ctrl_off + 4 * (128 + chunk), seq)
                for src in range(world):
                    if src == args.rank:
                        continue
                    length = (
                        (src + 1) * total_packs // world - src * total_packs // world
                    ) * 16
                    for chunk in range(chunks):
                        for rail in range(2):
                            wait(
                                layout.pipe_flag_off
                                + (((world + src) * 64 + chunk) * 2 + rail)
                                * layout.flag_stride,
                                seq,
                            )
                    start = layout.pipe_recv_off + (src * 2 + 1) * layout.shard_cap
                    assert memory[start : start + length] == bytes([reduced]) * length
                wait(layout.ctrl_off + 4 * 48, seq)
            else:
                sizes = tuple((0, 16, 131088, 1 << 20)[r % 4] for r in range(world))
                sizes = sizes[seq % world :] + sizes[: seq % world]
                size = sizes[args.rank]
                chunk_bytes = 65536
                chunks = max(1, (size + chunk_bytes - 1) // chunk_bytes)
                start = (
                    layout.varlen_recv_off
                    + (slot * world + args.rank) * layout.varlen_bytes
                )
                memory[start : start + size] = bytes([pattern]) * size
                ctrl[4 + slot], ctrl[8 + slot] = size, 2
                ctrl[10 + slot], ctrl[12 + slot] = chunk_bytes // 16, chunks
                for chunk in range(chunks):
                    put(layout.ctrl_off + 4 * (256 + chunk), seq)
                put(layout.ctrl_off, seq)
                for src, length in enumerate(sizes):
                    if src == args.rank:
                        continue
                    for chunk in range(
                        max(1, (length + chunk_bytes - 1) // chunk_bytes)
                    ):
                        for rail in range(2):
                            wait(
                                layout.varlen_flag_off
                                + (
                                    (((slot * world + src) * 512 + chunk) * 2 + rail)
                                    * layout.flag_stride
                                ),
                                seq,
                            )
                    start = (
                        layout.varlen_recv_off
                        + (slot * world + src) * layout.varlen_bytes
                    )
                    assert (
                        memory[start : start + length]
                        == bytes([(seq + 17 * src) % 256]) * length
                    )
        group.exchange("drained")
        proxy.stop()
        stats = proxy.stats()
        assert stats["connected_peers"] == (
            world - 1 if args.mesh else min(2, world - 1)
        )
        if not args.mesh and world > 3:
            assert stats["ring_forwarded"] > 0, stats
        print(
            json.dumps(
                {
                    "rank": args.rank,
                    "world": world,
                    "iterations": args.iterations,
                    "routes": routes,
                    "stats": stats,
                }
            ),
            flush=True,
        )
    finally:
        proxy.close()
        del ctrl, buffer
        memory.close()


if __name__ == "__main__":
    main()
