"""Transport configuration must preserve the fabric's DSCP/ECN marking."""

import pytest

from b12x.comm.roce._proxy import _traffic_class


@pytest.mark.parametrize(
    "override,nccl,expected",
    [(None, None, 0), (None, "106", 106), ("194", "106", 194),
     ("0", "106", 0), ("0x6a", None, 106), ("255", None, 255)],
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
