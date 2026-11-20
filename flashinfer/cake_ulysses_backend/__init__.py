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

Backend of the NVLink path of :class:`flashinfer.comm.UlyssesCommunicator`:
the Cake Ulysses sequence-parallel all-to-all (scatter / gather heads) for
single-node NVLink groups on SM100 / SM103 as one self-contained package (the
:mod:`.cake_backend` host handle, the :mod:`.cake_jit` registry, the
:mod:`.cake_build` loader and the generated CUDA sources under ``csrc/``).
The public API, topology decision and NCCL fallback live in
:mod:`flashinfer.comm.ulysses`; import that module.
"""

from __future__ import annotations

from . import cake_backend, cake_jit
from .cake_backend import CakeUlyssesA2A, route_arch
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    DTYPES,
    MODES,
    PEER_TABLE_CAPACITY,
    SIGNAL_BYTES,
    SIGNAL_WORDS,
    WORLD_SIZES,
    arch_for_capability,
    record_for,
    record_name,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "DTYPES",
    "MODES",
    "PEER_TABLE_CAPACITY",
    "SIGNAL_BYTES",
    "SIGNAL_WORDS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "WORLD_SIZES",
    "CakeUlyssesA2A",
    "arch_for_capability",
    "cake_backend",
    "cake_jit",
    "record_for",
    "record_name",
    "route_arch",
]
