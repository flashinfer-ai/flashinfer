# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

"""Host-side requirements for reading the locality-domain PTX register."""

from __future__ import annotations

import functools


@functools.cache
def _dsl_cuda_version() -> tuple[int, int]:
    try:
        from cutlass.base_dsl.version_info import CUDA_VERSION

        return int(CUDA_VERSION.major), int(CUDA_VERSION.minor)
    except (ImportError, AttributeError, TypeError, ValueError):
        return (0, 0)


@functools.cache
def _driver_version() -> int:
    try:
        from cuda.bindings import driver

        error, version = driver.cuDriverGetVersion()
        if int(error):
            return 0
        return int(version)
    except (ImportError, AttributeError, TypeError, ValueError):
        return 0


def locality_domain_register_unsupported_reason(device=None) -> str | None:
    """Return why ``%locality_domain_id`` is unavailable, or ``None``."""
    toolkit = _dsl_cuda_version()
    if toolkit < (13, 5):
        found = f"{toolkit[0]}.{toolkit[1]}" if toolkit != (0, 0) else "unknown"
        return (
            "%locality_domain_id requires a CUTLASS DSL built with CUDA 13.5 "
            f"or newer, but found {found}"
        )

    driver = _driver_version()
    if driver < 13050:
        found = str(driver) if driver else "unknown"
        return (
            "%locality_domain_id requires a CUDA 13.5-compatible driver "
            f"(cuDriverGetVersion >= 13050), but found {found}"
        )

    try:
        from flashinfer.locality_domain import (
            LocalityDomainError,
            locality_domain_count,
        )

        count = locality_domain_count(device)
    except (LocalityDomainError, ValueError) as exc:
        return str(exc)
    if count < 2:
        return (
            "%locality_domain_id scheduling requires at least two locality "
            f"domains, but the device exposes {count}"
        )
    return None


__all__ = ["locality_domain_register_unsupported_reason"]
