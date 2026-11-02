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

# Program source of the Cake W4A16 MegaMoE route.
#
# The programs ship as the in-tree Cake family ``cake_mega_moe_w4a16``: the
# generated loader ``flashinfer/jit/cake_mega_moe_w4a16.py`` publishes one
# ``CakeProgram`` constant per program (:data:`PROGRAM_IDENTIFIERS`:
# ``FUSED_N<n>[_CG2][_H<h_max>]`` and ``COMBINE``), each shipped for the
# exact targets it was rendered for. The route is available when the loader
# publishes every program the host plan enumerates (:data:`PROGRAM_IDENTIFIERS`:
# the fused program of every cell at every known pull-slot row, and the combine
# program); a missing program is named when it is loaded. A program table
# resolves an identifier
# to a :class:`ProgramHandle`: the binding's argument names in call order
# (ending in ``grid_x, grid_y, grid_z``) and the launch callable. A table other
# than the shipped loader is installed with :func:`set_programs`. The loader is
# imported inside the functions that use it.

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import torch

from .cake_plan import (
    COMBINE,
    FUSED,
    GRID_ARGS,
    KERNELS,
    PROGRAM_IDENTIFIERS,
    SUPPLIED_ARGS,
    Cell,
    program_identifier,
)

LOADER_MODULE = "flashinfer.jit.cake_mega_moe_w4a16"
#: Exact targets of the family; a device is admitted when the programs ship
#: for its capability.
TARGETS: Dict[Tuple[int, int], str] = {(10, 0): "sm_100a", (10, 3): "sm_103a"}


@dataclass(frozen=True)
class ProgramHandle:
    """One loaded program.

    ``args`` are the binding's argument names in call order, ending in
    :data:`GRID_ARGS`; ``launch(*values)`` performs the launch with the values
    in that order on the tvm-ffi environment stream. ``cluster`` is the launch
    cluster shape, ``cooperative`` whether every CTA of the grid must be
    resident, ``use_pdl`` whether the launch carries programmatic stream
    serialization, and ``max_active_clusters(device_index)`` the device's
    co-resident cluster capacity for the launch (``None`` = no occupancy
    query; the host falls back to one CTA per SM).

    A program that names an argument this host does not supply
    (:data:`SUPPLIED_ARGS`) is refused here, with the difference.
    """

    kernel: str
    identifier: str
    args: Tuple[str, ...]
    launch: Callable[..., None]
    cluster: Tuple[int, int, int] = (1, 1, 1)
    cooperative: bool = False
    use_pdl: bool = True
    max_active_clusters: Optional[Callable[[int], int]] = None

    def __post_init__(self) -> None:
        if self.kernel not in KERNELS:
            raise RuntimeError(
                f"{self.identifier}: unknown kernel {self.kernel!r}; the route has {KERNELS}"
            )
        if tuple(self.args[-3:]) != GRID_ARGS:
            raise RuntimeError(
                f"{self.identifier}: the binding's arguments end with "
                f"{list(self.args[-3:])}, not {list(GRID_ARGS)}"
            )
        unknown = [
            name for name in self.kernel_args if name not in SUPPLIED_ARGS[self.kernel]
        ]
        if unknown:
            raise RuntimeError(
                f"{self.identifier}: the program takes {unknown}, which this host does "
                f"not supply; it supplies {list(SUPPLIED_ARGS[self.kernel][:-3])}"
            )
        if len(set(self.args)) != len(self.args):
            raise RuntimeError(
                f"{self.identifier}: duplicate argument names in {self.args}"
            )

    @property
    def kernel_args(self) -> Tuple[str, ...]:
        """The arguments before the grid triple."""

        return tuple(self.args[: -len(GRID_ARGS)])


class ProgramTable(Protocol):
    """Where the backend takes its programs from."""

    def handle(
        self,
        kernel: str,
        cell: Optional[Cell],
        device: torch.device,
        hidden: Optional[int] = None,
    ) -> ProgramHandle: ...

    def targets(self) -> Tuple[Tuple[int, int], ...]:
        """Compute capabilities the table serves."""
        ...


class ShippedPrograms:
    """The generated loader: ``CakeProgram`` constants resolved to the
    device's exact target, loaded once per (identifier, target)."""

    def __init__(self) -> None:
        self._handles: Dict[Tuple[str, str], ProgramHandle] = {}

    @staticmethod
    @functools.cache
    def loader() -> Any:
        return importlib.import_module(LOADER_MODULE)

    @classmethod
    def available(cls) -> bool:
        try:
            loader = cls.loader()
        except ImportError:
            return False
        return all(hasattr(loader, identifier) for identifier in PROGRAM_IDENTIFIERS)

    def targets(self) -> Tuple[Tuple[int, int], ...]:
        try:
            loader = self.loader()
        except ImportError:
            return ()
        shipped: Optional[set] = None
        for identifier in PROGRAM_IDENTIFIERS:
            program = getattr(loader, identifier, None)
            if program is None:
                return ()
            targets = set(program.targets)
            shipped = targets if shipped is None else shipped & targets
        return tuple(
            capability
            for capability, target in TARGETS.items()
            if shipped and target in shipped
        )

    def handle(
        self,
        kernel: str,
        cell: Optional[Cell],
        device: torch.device,
        hidden: Optional[int] = None,
    ) -> ProgramHandle:
        from flashinfer.utils import get_compute_capability

        identifier = program_identifier(kernel, cell, hidden)
        program = getattr(self.loader(), identifier, None)
        if program is None:
            raise RuntimeError(
                f"{LOADER_MODULE} publishes no program {identifier!r}; the route needs "
                f"every program of {list(PROGRAM_IDENTIFIERS)}"
            )
        capability = get_compute_capability(device)
        target = program.target_for(capability)
        if target is None:
            raise RuntimeError(
                "the Cake W4A16 MegaMoE route requires an exact sm_100a or sm_103a "
                f"build; {identifier} ships {list(program.targets)} and the device is "
                f"{capability[0]}.{capability[1]}"
            )
        key = (identifier, target)
        handle = self._handles.get(key)
        if handle is None:
            record = program.launch_record()
            expected_cluster = (
                (cell.cta_group, 1, 1) if kernel == FUSED and cell else (1, 1, 1)
            )
            if tuple(program.cluster) != expected_cluster:
                raise RuntimeError(
                    f"{identifier}: expected cluster {expected_cluster}, the loader "
                    f"records {tuple(program.cluster)}"
                )
            if not record["use_pdl"]:
                raise RuntimeError(
                    f"{identifier}: the Cake W4A16 MegaMoE programs launch with PDL; "
                    f"the loader records {record}"
                )
            if (kernel == FUSED) != bool(record["cooperative"]):
                raise RuntimeError(
                    f"{identifier}: the fused program launches cooperatively and the "
                    f"combine program does not; the loader records {record}"
                )
            module = program.load(target)
            handle = ProgramHandle(
                kernel=kernel,
                identifier=identifier,
                args=tuple(program.args),
                launch=module.run,
                cluster=tuple(program.cluster),
                cooperative=bool(record["cooperative"]),
                use_pdl=bool(record["use_pdl"]),
                max_active_clusters=getattr(module, "max_active_clusters", None),
            )
            self._handles[key] = handle
        return handle


_programs: Optional[ProgramTable] = None


def set_programs(table: Optional[ProgramTable]) -> None:
    """Install the program table (``None`` restores the default resolution)."""

    global _programs
    _programs = table


def programs() -> ProgramTable:
    """The installed table or the shipped loader."""

    global _programs
    if _programs is None:
        _programs = ShippedPrograms()
    return _programs


def programs_available() -> bool:
    """Whether a program table is installed or shipped."""

    if _programs is not None:
        return True
    return ShippedPrograms.available()


def admitted_capabilities() -> Tuple[Tuple[int, int], ...]:
    """Compute capabilities the current program table ships programs for."""

    try:
        return tuple(programs().targets())
    except Exception:  # noqa: BLE001 - admission is advisory; loading reports the cause
        return ()


def combine_handle(hidden: int, device: torch.device) -> ProgramHandle:
    return programs().handle(COMBINE, None, device, hidden)


def fused_handle(cell: Cell, hidden: int, device: torch.device) -> ProgramHandle:
    return programs().handle(FUSED, cell, device, hidden)


__all__ = [
    "LOADER_MODULE",
    "TARGETS",
    "ProgramHandle",
    "ProgramTable",
    "ShippedPrograms",
    "admitted_capabilities",
    "combine_handle",
    "fused_handle",
    "programs",
    "programs_available",
    "set_programs",
]
