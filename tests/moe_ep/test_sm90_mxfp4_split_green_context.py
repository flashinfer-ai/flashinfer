# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CPU-only contracts for Hopper Green Context partition ownership.

These tests intentionally use the FlashInfer package boundary.  The fake
driver makes capability, validation, and cleanup behavior deterministic
without requiring a CUDA device or importing the donor repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
    GreenContextConfigurationError,
    GreenContextError,
    GreenContextSplit,
    check_green_context_support,
)


@dataclass
class _Sm:
    smCount: int
    minSmPartitionSize: int = 8
    smCoscheduledAlignment: int = 8


@dataclass
class _Resource:
    sm: _Sm


class _Handle:
    def __init__(self, kind: str, index: int) -> None:
        self.kind = kind
        self.index = index

    def __int__(self) -> int:
        return 0x1000 + self.index


class _FakeDriver:
    CUdevice_attribute = SimpleNamespace(
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR="major",
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR="minor",
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT="sm_count",
    )
    CUdevResourceType = SimpleNamespace(CU_DEV_RESOURCE_TYPE_SM="sm")
    CUgreenCtxCreate_flags = SimpleNamespace(CU_GREEN_CTX_DEFAULT_STREAM=1)
    CUstream_flags = SimpleNamespace(CU_STREAM_NON_BLOCKING=1)

    def __init__(
        self,
        *,
        fail_at: str | None = None,
        major: int = 9,
        minor: int = 0,
        sm_count: int = 132,
    ) -> None:
        self.fail_at = fail_at
        self.major = major
        self.minor = minor
        self.sm_count = sm_count
        self.calls: list[tuple] = []
        self._green_count = 0
        self._stream_count = 0

    def _result(self, name: str, *payload):
        self.calls.append((name, *payload))
        if self.fail_at == name:
            return (801,)
        return (0, *payload)

    def cuInit(self, _flags):
        return self._result("cuInit")

    def cuDeviceGet(self, ordinal):
        self.calls.append(("cuDeviceGet", ordinal))
        return (0, f"device:{ordinal}")

    def cuDeviceGetAttribute(self, attribute, device):
        self.calls.append(("cuDeviceGetAttribute", attribute, device))
        values = {
            "major": self.major,
            "minor": self.minor,
            "sm_count": self.sm_count,
        }
        return (0, values[attribute])

    def cuDeviceGetDevResource(self, device, resource_type):
        self.calls.append(("cuDeviceGetDevResource", device, resource_type))
        return (0, _Resource(_Sm(self.sm_count)))

    def cuDevSmResourceSplitByCount(self, groups, _resource, flags, minimum):
        self.calls.append(("cuDevSmResourceSplitByCount", groups, flags, minimum))
        return (
            0,
            [_Resource(_Sm(minimum))],
            1,
            _Resource(_Sm(self.sm_count - minimum)),
        )

    def cuDevResourceGenerateDesc(self, resources, _count):
        index = self._green_count + 1
        self.calls.append(("cuDevResourceGenerateDesc", resources[0].sm.smCount))
        if self.fail_at == f"cuDevResourceGenerateDesc:{index}":
            return (801,)
        return (0, _Handle("desc", index))

    def cuGreenCtxCreate(self, _descriptor, _device, flags):
        self._green_count += 1
        index = self._green_count
        self.calls.append(("cuGreenCtxCreate", index, flags))
        if self.fail_at == f"cuGreenCtxCreate:{index}":
            return (801,)
        return (0, _Handle("green", index))

    def cuCtxFromGreenCtx(self, green_context):
        index = green_context.index
        self.calls.append(("cuCtxFromGreenCtx", index))
        if self.fail_at == f"cuCtxFromGreenCtx:{index}":
            return (801,)
        return (0, _Handle("context", index))

    def cuGreenCtxStreamCreate(self, _green_context, flags, priority):
        self._stream_count += 1
        index = self._stream_count
        self.calls.append(("cuGreenCtxStreamCreate", index, flags, priority))
        if self.fail_at == f"cuGreenCtxStreamCreate:{index}":
            return (801,)
        return (0, _Handle("stream", index))

    def cuStreamDestroy(self, stream):
        self.calls.append(("cuStreamDestroy", stream.index))
        return (0,)

    def cuGreenCtxDestroy(self, green_context):
        self.calls.append(("cuGreenCtxDestroy", green_context.index))
        return (0,)


def _cleanup_calls(driver: _FakeDriver) -> list[tuple]:
    return [
        call
        for call in driver.calls
        if call[0] in ("cuStreamDestroy", "cuGreenCtxDestroy")
    ]


def test_capability_probe_records_h200_constraints() -> None:
    support = check_green_context_support(driver=_FakeDriver())

    assert support.supported
    assert support.reason is None
    assert support.compute_capability == (9, 0)
    assert support.total_sms == 132
    assert support.min_sm_partition_size == 8
    assert support.sm_coscheduled_alignment == 8


def test_h200_80_plus_52_split_is_disjoint_and_cleans_in_reverse() -> None:
    driver = _FakeDriver()
    with GreenContextSplit.create(80, driver=driver) as split:
        assert split.sm_counts == (80, 52)
        assert sum(split.sm_counts) == 132
        assert split.k1 is split.primary
        assert split.k2 is split.remainder
        assert split.k1.raw_green_context is not split.k2.raw_green_context
        assert split.k1.raw_stream is not split.k2.raw_stream
        assert [part.name for part in split.partitions] == ["k1", "k2"]
        assert [int(stream) for stream in split.raw_streams] == [0x1001, 0x1002]
        assert not split.closed

    assert split.closed
    assert _cleanup_calls(driver) == [
        ("cuStreamDestroy", 2),
        ("cuGreenCtxDestroy", 2),
        ("cuStreamDestroy", 1),
        ("cuGreenCtxDestroy", 1),
    ]
    split.close()
    assert len(_cleanup_calls(driver)) == 4


def test_external_stream_wrappers_are_lazy_cached_and_bound_to_device() -> None:
    driver = _FakeDriver()
    constructed: list[tuple[int, int]] = []

    class _ExternalStream:
        def __init__(self, pointer, *, device):
            constructed.append((pointer, device))
            self.pointer = pointer

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(ExternalStream=_ExternalStream))
    split = GreenContextSplit.create(80, device_ordinal=3, driver=driver)
    assert constructed == []

    assert split.torch_streams(fake_torch) == split.torch_streams(fake_torch)
    assert constructed == [(0x1001, 3), (0x1002, 3)]
    split.close()
    with pytest.raises(GreenContextError, match="already closed"):
        split.primary.as_torch_stream(fake_torch)


@pytest.mark.parametrize("invalid_count", [0, 7, 82, 132, 136, True, 80.0])
def test_invalid_partition_is_rejected_before_driver_split(invalid_count) -> None:
    driver = _FakeDriver()
    with pytest.raises(GreenContextConfigurationError):
        GreenContextSplit.create(invalid_count, driver=driver)
    assert not any(call[0] == "cuDevSmResourceSplitByCount" for call in driver.calls)


def test_second_stream_failure_cleans_all_created_handles_in_reverse() -> None:
    driver = _FakeDriver(fail_at="cuGreenCtxStreamCreate:2")
    with pytest.raises(GreenContextError, match="cuGreenCtxStreamCreate"):
        GreenContextSplit.create(80, driver=driver)

    assert _cleanup_calls(driver) == [
        ("cuGreenCtxDestroy", 2),
        ("cuStreamDestroy", 1),
        ("cuGreenCtxDestroy", 1),
    ]


def test_missing_binding_api_fails_capability_probe_closed() -> None:
    driver = _FakeDriver()
    driver.cuGreenCtxCreate = None
    support = check_green_context_support(driver=driver)

    assert not support.supported
    assert support.reason is not None
    assert "cuGreenCtxCreate" in support.reason


def test_non_hopper_capability_is_not_accepted_as_green_split() -> None:
    support = check_green_context_support(driver=_FakeDriver(major=8, sm_count=108))

    assert not support.supported
    assert support.reason
