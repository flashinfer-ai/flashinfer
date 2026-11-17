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


_NVCC_TARGET_PATTERN = re.compile(r"^(sm|compute)_([0-9]{2,3}[af]?)$")
_NVCC_GENCODE_FLAGS = ("-gencode", "--generate-code")
_NVCC_CODE_FLAGS = ("-code", "--gpu-code")
_NVCC_ARCH_FLAGS = ("-arch", "--gpu-architecture")


def cuda_targets_from_nvcc_flags(flags: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return the ``(SASS, PTX)`` targets that nvcc flags ask a module to embed.

    Understands ``-gencode=arch=...,code=...`` (also ``--generate-code`` and the
    two-token spelling), ``-code`` / ``--gpu-code`` and ``-arch`` /
    ``--gpu-architecture``. SASS targets come back normalized like the binary
    inspection does (``sm89``, ``sm120a``) and PTX targets as ``compute89``.
    ``-arch=sm_XX`` embeds both, as nvcc documents; ``-arch=native`` and the
    other non-numeric spellings request nothing statically known and are skipped.
    """
    flag_list = list(flags)
    sass: set[str] = set()
    ptx: set[str] = set()

    def add_code_list(code_list: str) -> None:
        for token in code_list.strip("[]'\" ").split(","):
            match = _NVCC_TARGET_PATTERN.fullmatch(token.strip("[]'\" "))
            if match is None:
                continue
            kind, arch = match.groups()
            (sass if kind == "sm" else ptx).add(
                f"sm{arch}" if kind == "sm" else f"compute{arch}"
            )

    for index, flag in enumerate(flag_list):
        name, separator, value = flag.partition("=")
        if name in _NVCC_GENCODE_FLAGS:
            if not separator:
                if index + 1 >= len(flag_list):
                    continue
                value = flag_list[index + 1]
            code_match = re.search(r"(?:^|,)code=(.*)$", value)
            if code_match is not None:
                add_code_list(code_match.group(1))
        elif name in _NVCC_CODE_FLAGS and separator:
            add_code_list(value)
        elif name in _NVCC_ARCH_FLAGS and separator:
            match = _NVCC_TARGET_PATTERN.fullmatch(value.strip())
            if match is None:
                continue
            kind, arch = match.groups()
            ptx.add(f"compute{arch}")
            if kind == "sm":
                sass.add(f"sm{arch}")
    return sorted(sass), sorted(ptx)


def provider_flag_issues(flags: Iterable[str], provider: str) -> list[str]:
    """Describe how a module's nvcc flags break the one-architecture provider rule.

    A provider wheel must contain only SASS its own architecture can run, and no
    PTX (the release validation inspects the built binaries with ``cuobjdump`` and
    rejects both). Returns an empty list when the flags request nothing explicit,
    since the default targets come from the already provider-filtered context.
    """
    sass, ptx = cuda_targets_from_nvcc_flags(flags)
    issues = []
    incompatible = [
        target
        for target in sass
        if cuda_binary_target_compatibility_score(target, provider) is None
    ]
    if incompatible:
        issues.append(f"emits {', '.join(incompatible)}, which {provider} cannot run")
    if ptx:
        issues.append(f"embeds PTX ({', '.join(ptx)}); provider wheels must not")
    return issues
