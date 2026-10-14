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

CUDA architecture compatibility helpers for JIT-cache providers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_CUDA_ARCHITECTURE_PATTERN = re.compile(r"^sm([0-9]{1,2})([0-9])([af]?)$")


@dataclass(frozen=True)
class CUDAArchitecture:
    major: int
    minor: int
    suffix: str


def parse_cuda_architecture(architecture: str) -> CUDAArchitecture:
    """Parse a normalized CUDA architecture such as ``sm90a``."""
    match = _CUDA_ARCHITECTURE_PATTERN.fullmatch(architecture)
    if match is None:
        raise ValueError(f"Invalid CUDA architecture {architecture!r}")
    return CUDAArchitecture(
        major=int(match.group(1)),
        minor=int(match.group(2)),
        suffix=match.group(3),
    )


def cuda_binary_target_compatibility_score(
    binary_target: str, device_architecture: str
) -> tuple[int, int] | None:
    """Return a preference score when a SASS target can run on a device.

    Architecture-specific ``a`` targets require an exact compute capability.
    Unsuffixed and family-specific ``f`` targets are forward compatible within
    their compute-capability major. Higher scores prefer an exact or more recent
    compatible target.
    """
    binary = parse_cuda_architecture(binary_target)
    device = parse_cuda_architecture(device_architecture)

    if (binary.major, binary.minor, binary.suffix) == (
        device.major,
        device.minor,
        device.suffix,
    ):
        return (4, binary.minor)
    if binary.suffix == "a":
        if (binary.major, binary.minor) == (device.major, device.minor):
            return (3, binary.minor)
        return None
    if binary.major != device.major or binary.minor > device.minor:
        return None
    return (2 if binary.suffix == "f" else 1, binary.minor)


def select_compatible_cuda_architecture(
    device_architecture: str, available_architectures: Iterable[str]
) -> str | None:
    """Select the best compatible target from an available provider set."""
    candidates = []
    for architecture in available_architectures:
        score = cuda_binary_target_compatibility_score(
            architecture, device_architecture
        )
        if score is not None:
            candidates.append((score, architecture))
    if not candidates:
        return None
    return max(candidates)[1]
