"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import torch
from packaging.version import Version

from .utils import get_compute_capability


SmSplit = Literal["backfill", "strict"]
_SM_SPLITS = ("backfill", "strict")
_DOMAIN_COUNT = 2
_REQUIRED_CUDA_VERSION = Version("13.4")


class LocalityDomainError(RuntimeError):
    """Raised when CUDA locality-domain memory or compute cannot be used."""


@dataclass(frozen=True)
class _DomainPools:
    pools: tuple[Any, Any]
    allocators: tuple[Any, Any]


@dataclass(frozen=True)
class _LocalizedStreams:
    streams: tuple[torch.cuda.ExternalStream, torch.cuda.ExternalStream]
    sm_counts: tuple[int, int]
    remainder_sm_count: int
    sm_coscheduled_alignment: int


@dataclass(frozen=True, eq=False)
class LocalizedDomains:
    """Matched locality-domain memory pools and green-context streams."""

    device: torch.device
    domain_count: int
    sm_split: SmSplit
    sm_counts: tuple[int, int]
    remainder_sm_count: int
    sm_coscheduled_alignment: int
    streams: tuple[torch.cuda.ExternalStream, torch.cuda.ExternalStream]
    pools: tuple[Any, Any]
    _module: Any = field(repr=False, compare=False)
    _allocators: tuple[Any, Any] = field(repr=False, compare=False)

    def empty(
        self,
        domain: int,
        shape: Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate a contiguous tensor in one device locality domain."""

        if domain not in range(_DOMAIN_COUNT):
            raise ValueError(f"locality domain must be 0 or 1, got {domain}")
        try:
            with _use_mem_pool()(self.pools[domain], device=self.device):
                return torch.empty(tuple(shape), dtype=dtype, device=self.device)
        except torch.OutOfMemoryError as exc:
            detail = str(self._module.last_error())
            suffix = f": {detail}" if detail else ""
            raise LocalityDomainError(
                f"allocation in CUDA locality domain {domain} failed{suffix}"
            ) from exc

    def pointer_domain(self, tensor: torch.Tensor) -> int:
        """Return the CUDA locality-domain ordinal backing ``tensor``."""

        if tensor.device != self.device:
            raise ValueError(f"tensor must be on {self.device}, got {tensor.device}")
        return int(self._module.pointer_domain(tensor.data_ptr()))


def _cuda_pluggable_allocator():
    return getattr(torch.cuda.memory, "CUDAPluggableAllocator", None)


def _mem_pool():
    return getattr(torch.cuda, "MemPool", None)


def _use_mem_pool():
    return getattr(torch.cuda, "use_mem_pool", None)


def _allocator_backend() -> str:
    get_backend = getattr(torch.cuda, "get_allocator_backend", None)
    if get_backend is None:
        get_backend = getattr(torch.cuda.memory, "get_allocator_backend", None)
    if get_backend is None:
        raise LocalityDomainError(
            "the installed PyTorch does not expose torch.cuda.get_allocator_backend"
        )
    try:
        return str(get_backend())
    except Exception as exc:
        raise LocalityDomainError(
            f"failed to query PyTorch's CUDA allocator backend: {exc}"
        ) from exc


def _get_cuda_version() -> Version:
    from .jit.cpp_ext import get_cuda_version

    return get_cuda_version()


def _normalize_device(device: torch.device | str | int | None) -> torch.device:
    if device is None:
        if not torch.cuda.is_available():
            raise LocalityDomainError("CUDA locality domains require a CUDA device")
        index = torch.cuda.current_device()
    else:
        resolved = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        if resolved.type != "cuda":
            raise LocalityDomainError(
                f"CUDA locality domains require a CUDA device, got {resolved}"
            )
        index = (
            torch.cuda.current_device()
            if resolved.index is None
            else int(resolved.index)
        )
    return torch.device("cuda", index)


def _validate_sm_split(sm_split: str) -> SmSplit:
    if sm_split not in _SM_SPLITS:
        raise ValueError(f"sm_split must be one of {_SM_SPLITS}, got {sm_split!r}")
    return cast(SmSplit, sm_split)


def _preflight(device: torch.device) -> None:
    # Resolve the device before compiling, so invalid ordinals fail cleanly.
    try:
        major, minor = get_compute_capability(device)
    except Exception as exc:
        raise LocalityDomainError(f"cannot access CUDA device {device}: {exc}") from exc
    if (major, minor) != (10, 7):
        raise LocalityDomainError(
            "CUDA locality domains are only supported on SM107 (Rubin); "
            f"device {device} is SM{major}{minor}"
        )
    if _cuda_pluggable_allocator() is None:
        raise LocalityDomainError(
            "the installed PyTorch does not expose "
            "torch.cuda.memory.CUDAPluggableAllocator"
        )
    if _mem_pool() is None or _use_mem_pool() is None:
        raise LocalityDomainError(
            "the installed PyTorch does not expose torch.cuda.MemPool and "
            "torch.cuda.use_mem_pool"
        )
    backend = _allocator_backend()
    if backend != "native":
        raise LocalityDomainError(
            "localized MemPool allocation requires PyTorch's native CUDA allocator; "
            f"the active backend is {backend!r}. Remove "
            "PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync"
        )
    try:
        cuda_version = _get_cuda_version()
    except Exception as exc:
        raise LocalityDomainError(
            f"cannot determine the CUDA toolkit version: {exc}"
        ) from exc
    if cuda_version < _REQUIRED_CUDA_VERSION:
        raise LocalityDomainError(
            "CUDA locality domains require CUDA 13.4 toolkit headers for JIT "
            f"compilation, but found CUDA {cuda_version}"
        )


@functools.cache
def _load_locality_module(device_index: int) -> tuple[Any, Path]:
    device = torch.device("cuda", device_index)
    _preflight(device)
    from .jit.locality_domain import gen_locality_domain_module

    spec = gen_locality_domain_module()
    try:
        module = spec.build_and_load()
    except Exception as exc:
        raise LocalityDomainError(
            "failed to JIT-build CUDA locality-domain support; CUDA 13.4 "
            f"toolkit headers and a working compiler are required: {exc}"
        ) from exc

    try:
        reason = str(module.probe(device_index))
    except Exception as exc:
        raise LocalityDomainError(
            f"CUDA locality-domain support probe failed: {exc}"
        ) from exc
    if reason:
        raise LocalityDomainError(reason)
    return module, spec.get_library_path().resolve()


@functools.cache
def _domain_pools(device_index: int) -> _DomainPools:
    module, library_path = _load_locality_module(device_index)
    device = torch.device("cuda", device_index)
    allocator_type = _cuda_pluggable_allocator()
    pool_type = _mem_pool()
    assert allocator_type is not None and pool_type is not None

    try:
        allocators = tuple(
            allocator_type(
                str(library_path),
                f"flashinfer_locality_domain{domain}_alloc",
                "flashinfer_locality_domain_free",
            )
            for domain in range(_DOMAIN_COUNT)
        )
        pools = tuple(pool_type(allocator.allocator()) for allocator in allocators)
    except Exception as exc:
        raise LocalityDomainError(
            f"failed to create locality-domain CUDA MemPools: {exc}"
        ) from exc

    # Verify the actual placement, not merely that the driver accepted the VMM
    # allocation property.
    for domain, pool in enumerate(pools):
        try:
            with _use_mem_pool()(pool, device=device):
                probe = torch.empty(1, dtype=torch.uint8, device=device)
        except torch.OutOfMemoryError as exc:
            detail = str(module.last_error())
            suffix = f": {detail}" if detail else ""
            raise LocalityDomainError(
                f"locality-domain {domain} allocation smoke test failed{suffix}"
            ) from exc
        try:
            actual_domain = int(module.pointer_domain(probe.data_ptr()))
        except Exception as exc:
            raise LocalityDomainError(
                f"failed to verify locality-domain {domain} allocation: {exc}"
            ) from exc
        if actual_domain != domain:
            raise LocalityDomainError(
                f"locality-domain {domain} allocation was placed in domain "
                f"{actual_domain}"
            )

    return _DomainPools(
        pools=cast(tuple[Any, Any], pools),
        allocators=cast(tuple[Any, Any], allocators),
    )


@functools.cache
def _localized_streams(
    device_index: int,
    sm_split: SmSplit,
) -> _LocalizedStreams:
    module, _ = _load_locality_module(device_index)
    try:
        values = tuple(
            int(value)
            for value in module.create_localized_streams(
                device_index, sm_split == "backfill"
            )
        )
    except Exception as exc:
        raise LocalityDomainError(
            f"failed to create {sm_split} locality-domain green contexts: {exc}"
        ) from exc
    if len(values) != 6:
        raise LocalityDomainError(
            "locality-domain stream setup returned invalid metadata"
        )
    stream0, stream1, sm0, sm1, remainder, sm_alignment = values
    device = torch.device("cuda", device_index)
    try:
        streams = (
            torch.cuda.ExternalStream(stream0, device=device),
            torch.cuda.ExternalStream(stream1, device=device),
        )
    except Exception as exc:
        raise LocalityDomainError(
            f"failed to wrap {sm_split} locality-domain CUDA streams: {exc}"
        ) from exc
    return _LocalizedStreams(
        streams=streams,
        sm_counts=(sm0, sm1),
        remainder_sm_count=remainder,
        sm_coscheduled_alignment=sm_alignment,
    )


@functools.cache
def _get_localized_domains(
    device_index: int,
    sm_split: SmSplit,
) -> LocalizedDomains:
    module, _ = _load_locality_module(device_index)
    try:
        domain_count = int(module.domain_count(device_index))
    except Exception as exc:
        raise LocalityDomainError(
            f"failed to query CUDA locality-domain count: {exc}"
        ) from exc
    if domain_count < 0:
        detail = str(module.last_error())
        suffix = f": {detail}" if detail else ""
        raise LocalityDomainError(f"failed to query CUDA locality-domain count{suffix}")
    if domain_count < _DOMAIN_COUNT:
        detail = str(module.last_error())
        suffix = f": {detail}" if detail else ""
        raise LocalityDomainError(
            f"device {device_index} exposes only {domain_count} locality domains"
            f"{suffix}"
        )
    streams = _localized_streams(device_index, sm_split)
    pools = _domain_pools(device_index)
    return LocalizedDomains(
        device=torch.device("cuda", device_index),
        domain_count=domain_count,
        sm_split=sm_split,
        sm_counts=streams.sm_counts,
        remainder_sm_count=streams.remainder_sm_count,
        sm_coscheduled_alignment=streams.sm_coscheduled_alignment,
        streams=streams.streams,
        pools=pools.pools,
        _module=module,
        _allocators=pools.allocators,
    )


def get_localized_domains(
    device: torch.device | str | int | None = None,
    *,
    sm_split: SmSplit = "backfill",
) -> LocalizedDomains:
    """Return process-lifetime locality-domain pools and compute streams.

    The default ``backfill`` split uses every SM, accepting stitch traffic for
    SMs borrowed from the other locality domain. ``strict`` uses only the SMs
    local to each domain and can leave the device's remainder SMs idle.
    """

    resolved = _normalize_device(device)
    split = _validate_sm_split(sm_split)
    assert resolved.index is not None
    return _get_localized_domains(resolved.index, split)


def locality_domain_unsupported_reason(
    device: torch.device | str | int | None = None,
    *,
    sm_split: SmSplit = "backfill",
) -> str | None:
    """Return why locality-domain support is unavailable, or ``None``."""

    try:
        get_localized_domains(device, sm_split=sm_split)
    except (LocalityDomainError, ValueError) as exc:
        return str(exc)
    return None


__all__ = [
    "LocalizedDomains",
    "LocalityDomainError",
    "SmSplit",
    "get_localized_domains",
    "locality_domain_unsupported_reason",
]
