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
