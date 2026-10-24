"""Transport configuration must preserve the fabric's DSCP/ECN marking."""

import pytest

from b12x.comm.roce._proxy import _traffic_class


@pytest.mark.parametrize(
    "override,nccl,expected",
    [
        (None, None, 0),
        (None, "106", 106),
        ("194", "106", 194),
        ("0", "106", 0),
        ("0x6a", None, 106),
        ("255", None, 255),
    ],
)
def test_traffic_class_precedence(monkeypatch, override, nccl, expected):
    for name, value in (("B12X_ROCE_TRAFFIC_CLASS", override), ("NCCL_IB_TC", nccl)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert _traffic_class() == expected


@pytest.mark.parametrize("name", ["B12X_ROCE_TRAFFIC_CLASS", "NCCL_IB_TC"])
@pytest.mark.parametrize("value", ["", "-1", "256", "106x", "1.5"])
def test_invalid_traffic_class_rejected(monkeypatch, name, value):
    monkeypatch.delenv("B12X_ROCE_TRAFFIC_CLASS", raising=False)
    monkeypatch.setenv("NCCL_IB_TC", "106")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        _traffic_class()


def test_route_table_matches_native_peer_rail_pair_order():
    from b12x.comm.roce._proxy import Proxy

    class Native:
        def roce_blob_bytes(self):
            return 4

        def roce_connect(self, ctx, blobs, size, table, rails):
            assert ctx == 123
            assert blobs.raw == b"aaaabbbbcccc"
            assert size == 12 and rails == 2
            assert list(table) == [0, 0, 0, 0, 0, 1, 2, 3, 1, 0, 3, 2]
            return 0

    proxy = Proxy.__new__(Proxy)
    proxy._ctx = 123
    proxy._lib = Native()
    proxy.world_size, proxy.rank = 3, 0
    try:
        proxy.connect(
            [b"aaaa", b"bbbb", b"cccc"], [[], [(0, 1), (2, 3)], [(1, 0), (3, 2)]]
        )
        with pytest.raises(RuntimeError, match="rails"):
            proxy.connect([b"aaaa", b"bbbb", b"cccc"], [[], [(0, 1)], [(1, 0), (3, 2)]])
    finally:
        proxy._ctx = None


def test_ring_route_table_leaves_non_neighbors_unconnected():
    from b12x.comm.roce._proxy import Proxy

    class Native:
        def roce_blob_bytes(self):
            return 4

        def roce_connect(self, ctx, blobs, size, table, rails):
            assert ctx == 123 and size == 16 and rails == 2
            assert list(table) == [0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 1, 0, 3, 2]
            return 0

    proxy = Proxy.__new__(Proxy)
    proxy._ctx, proxy._lib = 123, Native()
    proxy.world_size, proxy.rank, proxy.ring = 4, 0, True
    blobs = [b"aaaa", b"bbbb", b"cccc", b"dddd"]
    routes = [[], [(0, 1), (2, 3)], [], [(1, 0), (3, 2)]]
    try:
        proxy.connect(blobs, routes)
        routes[2] = [(0, 0), (1, 1)]
        with pytest.raises(RuntimeError, match="non-neighbors"):
            proxy.connect(blobs, routes)
    finally:
        proxy._ctx = None


def test_layout_places_two_shot_area_after_control():
    """The two-shot area follows the control record; without one the region is unchanged."""
    from b12x.comm.roce._proxy import Layout, load

    try:
        load()
    except RuntimeError as exc:  # no C compiler or libibverbs headers
        pytest.skip(str(exc))
    plain = Layout(4, 2 << 20)
    assert plain.pipe_bytes == 0 and plain.total_bytes == plain.pipe_send_off
    piped = Layout(4, 2 << 20, 64 << 20)
    assert piped.ctrl_off == plain.ctrl_off
    assert piped.pipe_send_off >= piped.ctrl_off + 4096
    assert piped.pipe_recv_off == piped.pipe_send_off + (64 << 20)
    assert piped.shard_cap == 16 << 20
    assert piped.pipe_flag_off == piped.pipe_recv_off + 4 * 2 * piped.shard_cap
    flags = 2 * 4 * piped.max_chunks * 2 * piped.flag_stride
    assert piped.total_bytes == piped.pipe_flag_off + flags
    # Uneven shards round up to whole pages.
    assert Layout(3, 2 << 20, 4096 * 5).shard_cap == 8192
    with pytest.raises(ValueError):
        Layout(4, 2 << 20, 1000)


def test_layout_places_variable_length_area_after_two_shot():
    """Two receive rows per rank and slot, then per-slot chunk flags; without one, nothing changes."""
    from b12x.comm.roce._proxy import ABI_VERSION, Layout, load

    try:
        lib = load()
    except RuntimeError as exc:  # no C compiler or libibverbs headers
        pytest.skip(str(exc))
    assert lib.roce_abi_version() == ABI_VERSION == 10
    piped = Layout(4, 2 << 20, 64 << 20)
    assert piped.varlen_bytes == 0
    assert piped.varlen_recv_off == piped.varlen_flag_off == piped.total_bytes
    varlen = Layout(4, 2 << 20, 64 << 20, 8 << 20)
    assert varlen.pipe_flag_off == piped.pipe_flag_off
    assert varlen.varlen_recv_off == piped.total_bytes
    assert varlen.varlen_flag_off == varlen.varlen_recv_off + 2 * 4 * (8 << 20)
    flags = 2 * 4 * varlen.varlen_max_chunks * 2 * varlen.flag_stride
    assert varlen.total_bytes == varlen.varlen_flag_off + flags
    # Without a two-shot area the variable-length one follows the control record.
    alone = Layout(2, 4096, 0, 4096)
    assert alone.varlen_recv_off == alone.pipe_send_off == alone.ctrl_off + 4096
    for bad in (1000, (1 << 32) + 4096):
        with pytest.raises(ValueError):
            Layout(4, 2 << 20, 0, bad)
