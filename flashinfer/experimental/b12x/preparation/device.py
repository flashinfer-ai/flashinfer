"""Lazy CUDA-device detection for preparation sessions."""

from __future__ import annotations

from dataclasses import dataclass

from .types import DeviceIdentity


@dataclass(frozen=True)
class DetectedDevice:
    ordinal: int | None
    identity: DeviceIdentity | None
    uuid: str | None = None
    max_shared_memory_per_block: int | None = None
    max_shared_memory_per_multiprocessor: int | None = None


_DEVICE_CACHE: dict[int, DeviceIdentity] = {}
_DEVICE_UUID_CACHE: dict[int, str] = {}
_DEVICE_SHARED_MEMORY_CACHE: dict[int, int] = {}
_DEVICE_MULTIPROCESSOR_SHARED_MEMORY_CACHE: dict[int, int] = {}


def detect_device(device: object | None = None) -> DetectedDevice:
    """Resolve one CUDA device without importing torch at package import."""

    try:
        import torch
    except ImportError:
        return DetectedDevice(ordinal=None, identity=None)
    if not torch.cuda.is_available():
        return DetectedDevice(ordinal=None, identity=None)
    resolved = torch.device("cuda" if device is None else device)
    if resolved.type != "cuda":
        return DetectedDevice(ordinal=None, identity=None)
    ordinal = resolved.index
    if ordinal is None:
        ordinal = int(torch.cuda.current_device())
    identity = _DEVICE_CACHE.get(ordinal)
    uuid = _DEVICE_UUID_CACHE.get(ordinal)
    max_shared_memory = _DEVICE_SHARED_MEMORY_CACHE.get(ordinal)
    max_multiprocessor_shared_memory = (
        _DEVICE_MULTIPROCESSOR_SHARED_MEMORY_CACHE.get(ordinal)
    )
    if (
        identity is None
        or uuid is None
        or max_shared_memory is None
        or max_multiprocessor_shared_memory is None
    ):
        properties = torch.cuda.get_device_properties(ordinal)
        if identity is None:
            identity = DeviceIdentity(
                vendor="nvidia",
                compute_capability=(
                    int(properties.major),
                    int(properties.minor),
                ),
                sm_count=int(properties.multi_processor_count),
                product_name=str(properties.name),
            )
            _DEVICE_CACHE[ordinal] = identity
        if uuid is None:
            uuid = str(properties.uuid).strip()
            if not uuid:
                raise RuntimeError("CUDA device UUID is unavailable")
            _DEVICE_UUID_CACHE[ordinal] = uuid
        if max_shared_memory is None:
            max_shared_memory = int(properties.shared_memory_per_block_optin)
            if max_shared_memory <= 0:
                raise RuntimeError("CUDA shared-memory limit is unavailable")
            _DEVICE_SHARED_MEMORY_CACHE[ordinal] = max_shared_memory
        if max_multiprocessor_shared_memory is None:
            max_multiprocessor_shared_memory = int(
                properties.shared_memory_per_multiprocessor
            )
            if max_multiprocessor_shared_memory <= 0:
                raise RuntimeError(
                    "CUDA multiprocessor shared-memory limit is unavailable"
                )
            _DEVICE_MULTIPROCESSOR_SHARED_MEMORY_CACHE[ordinal] = (
                max_multiprocessor_shared_memory
            )
    return DetectedDevice(
        ordinal=ordinal,
        identity=identity,
        uuid=uuid,
        max_shared_memory_per_block=max_shared_memory,
        max_shared_memory_per_multiprocessor=max_multiprocessor_shared_memory,
    )


__all__ = ["DetectedDevice", "detect_device"]
