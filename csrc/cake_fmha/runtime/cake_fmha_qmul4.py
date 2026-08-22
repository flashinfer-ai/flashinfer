# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed Blackwell QMUL4 cubin patch/load protocol.

The native CUDA source families contain register-faithful LOP3 markers because
public PTX does not expose the required QMUL4 lane-swizzle variants.  Compile a
native source to a cubin, then pass the cubin and the manifest-recorded marker
counts to :func:`patch_and_load_qmul4_cubin`.  Never load an unpatched native
cubin: the marker instruction is valid SASS but does not implement QMUL4.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping


QMUL4_MARKER_BASE = 0x51A7E000
QMUL4_RECIPES = {
    0: (7, 0),  # DIRECT: N6420 x B3210
    1: (7, 4),  # B0
    2: (7, 3),  # B1
    3: (7, 2),  # B2
    4: (7, 1),  # B3
    5: (6, 4),  # LOWER4
    6: (5, 4),  # HIGHER4
}
SUPPORTED_TARGETS = {"sm_100a": (10, 0), "sm_103a": (10, 3)}


def marker_counts_from_source(source: str) -> dict[int, int]:
    """Return the exact source-level marker denominator."""

    return {
        marker_id: source.count(f"0x{QMUL4_MARKER_BASE + marker_id:08X}")
        for marker_id in QMUL4_RECIPES
        if f"0x{QMUL4_MARKER_BASE + marker_id:08X}" in source
    }


def _validated_expected_counts(expected_counts: Mapping[int | str, int]) -> dict[int, int]:
    normalized = {int(marker_id): int(count) for marker_id, count in expected_counts.items()}
    if not normalized:
        raise RuntimeError("native QMUL4 loading requires nonempty expected marker counts")
    unknown = set(normalized).difference(QMUL4_RECIPES)
    if unknown:
        raise RuntimeError(f"unknown native QMUL4 marker ids: {sorted(unknown)}")
    invalid = {marker_id: count for marker_id, count in normalized.items() if count <= 0}
    if invalid:
        raise RuntimeError(f"native QMUL4 marker counts must be positive: {invalid}")
    return normalized


def patch_qmul4_cubin(
    cubin: bytes | bytearray | memoryview,
    *,
    expected_counts: Mapping[int | str, int],
) -> bytes:
    """Replace exact marker instructions and reject incomplete/drifted cubins."""

    expected = _validated_expected_counts(expected_counts)
    patched = bytearray(cubin)
    actual = {marker_id: 0 for marker_id in QMUL4_RECIPES}
    for marker_id, (a_swizzle, b_swizzle) in QMUL4_RECIPES.items():
        marker = QMUL4_MARKER_BASE + marker_id
        needle = struct.pack("<I", marker)
        search_from = 0
        while True:
            marker_offset = patched.find(needle, search_from)
            if marker_offset < 0:
                break
            instruction_offset = marker_offset - 4
            search_from = marker_offset + len(needle)
            if instruction_offset < 0 or instruction_offset + 16 > len(patched):
                continue
            low, high = struct.unpack_from("<QQ", patched, instruction_offset)
            if low >> 32 != marker or low & 0xFFFF != 0x7812:
                continue
            if high & 0xFFFFFF00 != 0x078E9600:
                continue
            rd = (low >> 16) & 0xFF
            ra = (low >> 24) & 0xFF
            rb = high & 0xFF
            qmul_low = (
                (b_swizzle << 59)
                | (rb << 32)
                | (ra << 24)
                | (rd << 16)
                | 0x727C
            )
            qmul_high = (high & 0xFFFFFFFF00000000) | 0x0501A000 | (a_swizzle << 10)
            struct.pack_into("<QQ", patched, instruction_offset, qmul_low, qmul_high)
            actual[marker_id] += 1

    mismatched = {
        marker_id: {"expected": count, "patched": actual[marker_id]}
        for marker_id, count in expected.items()
        if actual[marker_id] != count
    }
    unexpected = {
        marker_id: count
        for marker_id, count in actual.items()
        if count and marker_id not in expected
    }
    if mismatched or unexpected:
        raise RuntimeError(
            "native QMUL4 cubin patch count mismatch: "
            f"mismatched={mismatched}, unexpected={unexpected}; refusing to load"
        )
    return bytes(patched)


def patch_and_load_qmul4_cubin(
    cubin: bytes | bytearray | memoryview,
    *,
    expected_counts: Mapping[int | str, int],
    target: str,
):
    """Patch a cubin and load it into the current exact-target CUDA context."""

    if target not in SUPPORTED_TARGETS:
        raise RuntimeError(
            f"native QMUL4 target must be one of {sorted(SUPPORTED_TARGETS)}, got {target!r}"
        )
    patched = patch_qmul4_cubin(cubin, expected_counts=expected_counts)
    try:
        from cuda.bindings import driver
    except ImportError as exc:
        raise RuntimeError("native QMUL4 loading requires cuda-python") from exc

    (status,) = driver.cuInit(0)
    if status != 0:
        raise RuntimeError(f"cuInit failed before native QMUL4 load: CUresult={status}")
    status, device = driver.cuCtxGetDevice()
    if status != 0:
        raise RuntimeError(
            "native QMUL4 loading requires a current CUDA context: "
            f"cuCtxGetDevice CUresult={status}"
        )
    attribute = driver.CUdevice_attribute
    status_major, major = driver.cuDeviceGetAttribute(
        attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
    )
    status_minor, minor = driver.cuDeviceGetAttribute(
        attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device
    )
    if status_major != 0 or status_minor != 0:
        raise RuntimeError(
            "failed to resolve the current device architecture before native QMUL4 load"
        )
    expected_cc = SUPPORTED_TARGETS[target]
    if (int(major), int(minor)) != expected_cc:
        raise RuntimeError(
            f"native QMUL4 cubin target {target} requires CC {expected_cc[0]}.{expected_cc[1]}, "
            f"current device is CC {major}.{minor}"
        )
    status, module = driver.cuModuleLoadData(patched)
    if status != 0:
        raise RuntimeError(f"cuModuleLoadData failed for patched native QMUL4 cubin: CUresult={status}")
    return module


__all__ = [
    "QMUL4_MARKER_BASE",
    "QMUL4_RECIPES",
    "SUPPORTED_TARGETS",
    "marker_counts_from_source",
    "patch_and_load_qmul4_cubin",
    "patch_qmul4_cubin",
]
